using System;
using System.Collections.Generic;
using System.Text;
using System.Xml.Serialization;
using System.Xml;
using System.IO;

namespace DunefieldModel
{
  public class Universal {
    private static Dictionary<Type, XmlSerializer> serializers = new Dictionary<Type, XmlSerializer>();

    public static XmlSerializer getSerializer(Type ForType) {
      if (!serializers.ContainsKey(ForType))
        serializers.Add(ForType, new XmlSerializer(ForType));
      return serializers[ForType];
    }

    public static object XmlToObj(Type objType, string buf){
      XmlReader xr = XmlReader.Create(new StringReader(buf));
      object o = null;
      if (buf.Length > 0)
        try {
          o = getSerializer(objType).Deserialize(xr);
        } catch { }
      return o;
    }

    public static string ObjToXml(object obj) {
      StringBuilder sb = new StringBuilder();
      if (obj != null) {
        XmlSerializerNamespaces ns = new XmlSerializerNamespaces();
        ns.Add("", "");
        XmlWriterSettings xrs = new XmlWriterSettings();
        xrs.OmitXmlDeclaration = true;
        XmlWriter xw = XmlWriter.Create(sb, xrs);
        getSerializer(obj.GetType()).Serialize(xw, obj, ns);
        xw.Flush();
      }
      return sb.ToString();
    }

    public static int TimeZoneOffsetFromUtc = -6;
    public static DateTimeOffset AddOffset(DateTime TimeAtLocation) {
      return new DateTimeOffset(TimeAtLocation, new TimeSpan(TimeZoneOffsetFromUtc, 0, 0));
    }

    public static DateTime? ToDateTimeN(DateTimeOffset? DTO) {
      return DTO.HasValue ? DTO.Value.DateTime : (DateTime?)null;
    }

    public const char UidFieldSeparator = '=';

    public static string Null2str(object obj) {
      return (obj == null) ? "" : obj.ToString();
    }

  }
}
